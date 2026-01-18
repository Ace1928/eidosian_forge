from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
@memoize
def get_expansion_candidate_drives(self):
    """Retrieve required expansion drive list.

        Note: To satisfy the expansion criteria each item in the candidate list must added specified group since there
        is a potential limitation on how many drives can be incorporated at a time.
            * Traditional raid volume groups must be added two drives maximum at a time. No limits on raid disk pools.

        :return list(candidate): list of candidate structures from the getVolumeGroupExpansionCandidates symbol endpoint
        """

    def get_expansion_candidate_drive_request():
        """Perform the request for expanding existing volume groups or disk pools.

            Note: the list of candidate structures do not necessarily produce candidates that meet all criteria.
            """
        candidates_list = None
        url = 'storage-systems/%s/symbol/getVolumeGroupExpansionCandidates?verboseErrorResponse=true' % self.ssid
        if self.raid_level == 'raidDiskPool':
            url = 'storage-systems/%s/symbol/getDiskPoolExpansionCandidates?verboseErrorResponse=true' % self.ssid
        try:
            rc, candidates_list = self.request(url, method='POST', data=self.pool_detail['id'])
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve volume candidates. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        return candidates_list['candidates']
    required_candidate_list = list()
    required_additional_drives = 0
    required_additional_capacity = 0
    total_required_capacity = 0
    if self.criteria_min_usable_capacity:
        total_required_capacity = self.criteria_min_usable_capacity
        required_additional_capacity = self.criteria_min_usable_capacity - int(self.pool_detail['totalRaidedSpace'])
    if self.criteria_drive_count:
        required_additional_drives = self.criteria_drive_count - len(self.storage_pool_drives)
    if required_additional_drives > 0 or required_additional_capacity > 0:
        for candidate in get_expansion_candidate_drive_request():
            if self.criteria_drive_min_size:
                if self.criteria_drive_min_size > min(self.get_available_drive_capacities(candidate['drives'])):
                    continue
            if self.raid_level == 'raidDiskPool':
                if len(candidate['drives']) >= required_additional_drives and self.get_ddp_capacity(candidate['drives']) >= total_required_capacity:
                    required_candidate_list.append(candidate)
                    break
            else:
                required_additional_drives -= len(candidate['drives'])
                required_additional_capacity -= int(candidate['usableCapacity'])
                required_candidate_list.append(candidate)
            if required_additional_drives <= 0 and required_additional_capacity <= 0:
                break
        else:
            self.module.fail_json(msg='Not enough drives to meet the specified criteria. Array [%s].' % self.ssid)
    return required_candidate_list