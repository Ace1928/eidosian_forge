from __future__ import absolute_import, division, print_function
import os
import hashlib
from time import sleep
from threading import Thread
from ansible.module_utils.urls import open_url
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def download_device_files(self, headers, temp_target_disk, device_url, lease_updater, total_bytes_written, total_bytes_to_write):
    mf_content = 'SHA256(' + os.path.basename(temp_target_disk) + ')= '
    sha256_hash = hashlib.sha256()
    response = None
    with open(self.mf_file, 'a') as mf_handle:
        with open(temp_target_disk, 'wb') as handle:
            try:
                response = open_url(device_url, headers=headers, validate_certs=False, timeout=self.download_timeout)
            except Exception as err:
                lease_updater.httpNfcLease.HttpNfcLeaseAbort()
                lease_updater.stop()
                self.module.fail_json(msg='Exception caught when getting %s, %s' % (device_url, to_text(err)))
            if not response:
                lease_updater.httpNfcLease.HttpNfcLeaseAbort()
                lease_updater.stop()
                self.module.fail_json(msg='Getting %s failed' % device_url)
            if response.getcode() >= 400:
                lease_updater.httpNfcLease.HttpNfcLeaseAbort()
                lease_updater.stop()
                self.module.fail_json(msg='Getting %s return code %d' % (device_url, response.getcode()))
            current_bytes_written = 0
            block = response.read(self.chunk_size)
            while block:
                handle.write(block)
                sha256_hash.update(block)
                handle.flush()
                os.fsync(handle.fileno())
                current_bytes_written += len(block)
                block = response.read(self.chunk_size)
            written_percent = (current_bytes_written + total_bytes_written) * 100 / total_bytes_to_write
            lease_updater.progressPercent = int(written_percent)
        mf_handle.write(mf_content + sha256_hash.hexdigest() + '\n')
    self.facts['device_files'].append(temp_target_disk)
    return current_bytes_written