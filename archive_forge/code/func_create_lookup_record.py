from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def create_lookup_record(self, provider_addr, customer_addr, mac, vsid):
    lrec = self._scimv2.MSFT_NetVirtualizationLookupRecordSettingData(CustomerAddress=customer_addr, VirtualSubnetID=vsid)
    if lrec and lrec[0].VirtualSubnetID == vsid and (lrec[0].ProviderAddress == provider_addr) and (lrec[0].MACAddress == mac):
        return
    if lrec:
        lrec[0].Delete_()
    if constants.IPV4_DEFAULT == customer_addr:
        record_type = self._LOOKUP_RECORD_TYPE_L2_ONLY
    else:
        record_type = self._LOOKUP_RECORD_TYPE_STATIC
    self._create_new_object(self._scimv2.MSFT_NetVirtualizationLookupRecordSettingData, VirtualSubnetID=vsid, Rule=self._TRANSLATE_ENCAP, Type=record_type, MACAddress=mac, CustomerAddress=customer_addr, ProviderAddress=provider_addr)