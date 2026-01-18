from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def list_driver_vendor_passthru(self, driver):
    """Get driver's vendor_passthru methods.

        :param driver: The value can be the name of a driver or a
            :class:`~openstack.baremetal.v1.driver.Driver` instance.

        :returns: One :dict: of vendor methods with corresponding usages
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            driver matching the name could be found.
        """
    driver = self.get_driver(driver)
    return driver.list_vendor_passthru(self)