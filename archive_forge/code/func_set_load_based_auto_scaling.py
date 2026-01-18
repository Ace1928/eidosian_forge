import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def set_load_based_auto_scaling(self, layer_id, enable=None, up_scaling=None, down_scaling=None):
    """
        Specify the load-based auto scaling configuration for a
        specified layer. For more information, see `Managing Load with
        Time-based and Load-based Instances`_.


        To use load-based auto scaling, you must create a set of load-
        based auto scaling instances. Load-based auto scaling operates
        only on the instances from that set, so you must ensure that
        you have created enough instances to handle the maximum
        anticipated load.


        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type layer_id: string
        :param layer_id: The layer ID.

        :type enable: boolean
        :param enable: Enables load-based auto scaling for the layer.

        :type up_scaling: dict
        :param up_scaling: An `AutoScalingThresholds` object with the upscaling
            threshold configuration. If the load exceeds these thresholds for a
            specified amount of time, AWS OpsWorks starts a specified number of
            instances.

        :type down_scaling: dict
        :param down_scaling: An `AutoScalingThresholds` object with the
            downscaling threshold configuration. If the load falls below these
            thresholds for a specified amount of time, AWS OpsWorks stops a
            specified number of instances.

        """
    params = {'LayerId': layer_id}
    if enable is not None:
        params['Enable'] = enable
    if up_scaling is not None:
        params['UpScaling'] = up_scaling
    if down_scaling is not None:
        params['DownScaling'] = down_scaling
    return self.make_request(action='SetLoadBasedAutoScaling', body=json.dumps(params))