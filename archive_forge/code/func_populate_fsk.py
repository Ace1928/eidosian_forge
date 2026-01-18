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
def populate_fsk(self, fsk_filepath, fsk_pairs):
    """Writes in the fsk file all the substitution strings and their

        values which will populate the unattended file used when
        creating the pdk.
        """
    fabric_data_pairs = []
    for fsk_key, fsk_value in fsk_pairs.items():
        fabricdata = self._conn_msps.Msps_FabricData.new()
        fabricdata.key = fsk_key
        fabricdata.Value = fsk_value
        fabric_data_pairs.append(fabricdata)
    fsk = self._conn_msps.Msps_FSK.new()
    fsk.FabricDataPairs = fabric_data_pairs
    msps_pfp = self._conn_msps.Msps_ProvisioningFileProcessor
    msps_pfp.SerializeToFile(fsk_filepath, fsk)