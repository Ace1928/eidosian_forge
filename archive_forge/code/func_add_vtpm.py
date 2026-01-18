import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def add_vtpm(self, vm_name, pdk_filepath, shielded):
    """Adds a vtpm and enables it with encryption or shielded option."""
    vm = self._lookup_vm_check(vm_name)
    msps_pfp = self._conn_msps.Msps_ProvisioningFileProcessor
    provisioning_file = msps_pfp.PopulateFromFile(pdk_filepath)[0]
    key_protector = provisioning_file.KeyProtector
    policy_data = provisioning_file.PolicyData
    security_profile = _wqlutils.get_element_associated_class(self._conn, self._SECURITY_SETTING_DATA, element_uuid=vm.ConfigurationID)[0]
    security_profile.EncryptStateAndVmMigrationTraffic = True
    security_profile.TpmEnabled = True
    security_profile.ShieldingRequested = shielded
    sec_profile_serialized = security_profile.GetText_(1)
    job_path, ret_val = self._sec_svc.SetKeyProtector(key_protector, sec_profile_serialized)
    self._jobutils.check_ret_val(ret_val, job_path)
    job_path, ret_val = self._sec_svc.SetSecurityPolicy(policy_data, sec_profile_serialized)
    self._jobutils.check_ret_val(ret_val, job_path)
    job_path, ret_val = self._sec_svc.ModifySecuritySettings(sec_profile_serialized)
    self._jobutils.check_ret_val(ret_val, job_path)