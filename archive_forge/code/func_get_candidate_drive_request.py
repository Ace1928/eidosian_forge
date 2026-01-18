from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_candidate_drive_request():
    """Perform request for new volume creation."""
    candidates_list = list()
    drive_types = [self.criteria_drive_type] if self.criteria_drive_type else self.available_drive_types
    interface_types = [self.criteria_drive_interface_type] if self.criteria_drive_interface_type else self.available_drive_interface_types
    for interface_type in interface_types:
        for drive_type in drive_types:
            candidates = None
            volume_candidate_request_data = dict(type='diskPool' if self.raid_level == 'raidDiskPool' else 'traditional', diskPoolVolumeCandidateRequestData=dict(reconstructionReservedDriveCount=65535))
            candidate_selection_type = dict(candidateSelectionType='count', driveRefList=dict(driveRef=self.available_drives))
            criteria = dict(raidLevel=self.raid_level, phyDriveType=interface_type, dssPreallocEnabled=False, securityType='capable' if self.criteria_drive_require_fde else 'none', driveMediaType=drive_type, onlyProtectionInformationCapable=True if self.criteria_drive_require_da else False, volumeCandidateRequestData=volume_candidate_request_data, allocateReserveSpace=False, securityLevel='fde' if self.criteria_drive_require_fde else 'none', candidateSelectionType=candidate_selection_type)
            try:
                rc, candidates = self.request('storage-systems/%s/symbol/getVolumeCandidates?verboseErrorResponse=true' % self.ssid, data=criteria, method='POST')
            except Exception as error:
                self.module.fail_json(msg='Failed to retrieve volume candidates. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
            if candidates:
                candidates_list.extend(candidates['volumeCandidate'])
    tray_drawer_protection = list()
    tray_protection = list()
    drawer_protection = list()
    no_protection = list()
    sorted_candidates = list()
    for item in candidates_list:
        if item['trayLossProtection']:
            if item['drawerLossProtection']:
                tray_drawer_protection.append(item)
            else:
                tray_protection.append(item)
        elif item['drawerLossProtection']:
            drawer_protection.append(item)
        else:
            no_protection.append(item)
    if tray_drawer_protection:
        sorted_candidates.extend(tray_drawer_protection)
    if tray_protection:
        sorted_candidates.extend(tray_protection)
    if drawer_protection:
        sorted_candidates.extend(drawer_protection)
    if no_protection:
        sorted_candidates.extend(no_protection)
    return sorted_candidates