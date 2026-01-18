from pprint import pformat
from six import iteritems
import re
@rolling_update.setter
def rolling_update(self, rolling_update):
    """
        Sets the rolling_update of this V1beta2DeploymentStrategy.
        Rolling update config params. Present only if DeploymentStrategyType =
        RollingUpdate.

        :param rolling_update: The rolling_update of this
        V1beta2DeploymentStrategy.
        :type: V1beta2RollingUpdateDeployment
        """
    self._rolling_update = rolling_update