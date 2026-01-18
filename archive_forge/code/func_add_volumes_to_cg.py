from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def add_volumes_to_cg(self, cg_name, volumes, tiering_policy):
    """Add volumes to consistency group.
            :param cg_name: The name of the consistency group
            :param volumes: The list of volumes to be added to consistency
             group
            :param tiering_policy: The tiering policy that is to be applied to
            consistency group
            :return: The boolean value to indicate if volumes are added to
             consistency group
        """
    cg_details = self.unity_conn.get_cg(name=cg_name)._get_properties()
    existing_volumes_in_cg = cg_details['luns']
    existing_vol_ids = []
    if existing_volumes_in_cg:
        existing_vol_ids = [vol['UnityLun']['id'] for vol in existing_volumes_in_cg['UnityLunList']]
    ids_to_add = []
    vol_name_list = []
    vol_id_list = []
    all_vol_ids = []
    for vol in volumes:
        if 'vol_id' in vol and (not vol['vol_id'] in vol_id_list):
            vol_id_list.append(vol['vol_id'])
        elif 'vol_name' in vol and (not vol['vol_name'] in vol_name_list):
            vol_name_list.append(vol['vol_name'])
    'add volume by name'
    for vol in vol_name_list:
        ids_to_add.append(self.get_volume_details(vol_name=vol))
    'add volume by id'
    for vol in vol_id_list:
        'verifying if volume id exists in array'
        ids_to_add.append(self.get_volume_details(vol_id=vol))
    all_vol_ids = ids_to_add + existing_vol_ids
    ids_to_add = list(set(all_vol_ids) - set(existing_vol_ids))
    LOG.info('Volume IDs to add %s', ids_to_add)
    if len(ids_to_add) == 0:
        return False
    vol_add_list = []
    for vol in ids_to_add:
        vol_dict = {'id': vol}
        vol_add_list.append(vol_dict)
    cg_obj = self.return_cg_instance(cg_name)
    policy_enum = None
    if tiering_policy:
        if utils.TieringPolicyEnum[tiering_policy]:
            policy_enum = utils.TieringPolicyEnum[tiering_policy]
        else:
            errormsg = 'Invalid choice {0} for tiering policy'.format(tiering_policy)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
    try:
        cg_obj.modify(lun_add=vol_add_list, tiering_policy=policy_enum)
        return True
    except Exception as e:
        errormsg = 'Add existing volumes to consistency group {0} failed with error {1}'.format(cg_name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)