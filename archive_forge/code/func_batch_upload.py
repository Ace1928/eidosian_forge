from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def batch_upload(self):

    def _glob_files_locally(folder_path):
        len_folder_path = len(folder_path) + 1
        for root, v, files in os.walk(folder_path):
            for f in files:
                full_path = os.path.join(root, f)
                yield (full_path, full_path[len_folder_path:])

    def _normalize_blob_file_path(path, name):
        path_sep = '/'
        if path:
            name = path_sep.join((path, name))
        return path_sep.join(os.path.normpath(name).split(os.path.sep)).strip(path_sep)

    def _guess_content_type(file_path, original):
        if original.content_encoding or original.content_type:
            return original
        mimetypes.add_type('application/json', '.json')
        mimetypes.add_type('application/javascript', '.js')
        mimetypes.add_type('application/wasm', '.wasm')
        content_type, v = mimetypes.guess_type(file_path)
        return ContentSettings(content_type=content_type, content_disposition=original.content_disposition, content_language=original.content_language, content_md5=original.content_md5, cache_control=original.cache_control)
    if not os.path.exists(self.batch_upload_src):
        self.fail('batch upload source source directory {0} does not exist'.format(self.batch_upload_src))
    if not os.path.isdir(self.batch_upload_src):
        self.fail('incorrect usage: {0} is not a directory'.format(self.batch_upload_src))
    source_dir = os.path.realpath(self.batch_upload_src)
    source_files = list(_glob_files_locally(source_dir))
    content_settings = ContentSettings(content_type=self.content_type, content_encoding=self.content_encoding, content_language=self.content_language, content_disposition=self.content_disposition, cache_control=self.cache_control, content_md5=None)
    for src, blob_path in source_files:
        if self.batch_upload_dst:
            blob_path = _normalize_blob_file_path(self.batch_upload_dst, blob_path)
        if not self.check_mode:
            try:
                client = self.blob_service_client.get_blob_client(container=self.container, blob=blob_path)
                with open(src, 'rb') as data:
                    client.upload_blob(data=data, blob_type=self.get_blob_type(self.blob_type), metadata=self.tags, content_settings=_guess_content_type(src, content_settings), overwrite=self.force)
            except Exception as exc:
                self.fail('Error creating blob {0} - {1}'.format(src, str(exc)))
        self.results['actions'].append('created blob from {0}'.format(src))
    self.results['changed'] = True
    self.results['container'] = self.container_obj