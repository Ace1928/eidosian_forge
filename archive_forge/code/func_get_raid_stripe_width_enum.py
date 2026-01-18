from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_raid_stripe_width_enum(self, stripe_width):
    """ Get raid_stripe_width enum.
             :param stripe_width: The raid_stripe_width
             :return: raid_stripe_width enum
        """
    if stripe_width != 'BEST_FIT':
        stripe_width = '_' + stripe_width
    if stripe_width in utils.RaidStripeWidthEnum.__members__:
        return utils.RaidStripeWidthEnum[stripe_width]
    else:
        errormsg = 'Invalid choice %s for stripe width' % stripe_width
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)