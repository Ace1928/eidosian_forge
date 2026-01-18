from __future__ import absolute_import, division, print_function
import os
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
def determine_file_paths(self):
    """Determine all the drive firmware file paths."""
    self.files = {}
    if self.firmware:
        for firmware_path in self.firmware:
            if not os.path.exists(firmware_path):
                self.module.fail_json(msg='Drive firmware file does not exist! File [%s]' % firmware_path)
            elif os.path.isdir(firmware_path):
                if not firmware_path.endswith('/'):
                    firmware_path = firmware_path + '/'
                for dir_filename in os.listdir(firmware_path):
                    if '.dlp' in dir_filename:
                        self.files.update({dir_filename: firmware_path + dir_filename})
            elif '.dlp' in firmware_path:
                name = os.path.basename(firmware_path)
                self.files.update({name: firmware_path})