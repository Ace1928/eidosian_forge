from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
from ansible.module_utils.common.text.converters import to_native
def archived_image_manifest(archive_path):
    """
    Attempts to get Image.Id and image name from metadata stored in the image
    archive tar file.

    The tar should contain a file "manifest.json" with an array with a single entry,
    and the entry should have a Config field with the image ID in its file name, as
    well as a RepoTags list, which typically has only one entry.

    :raises:
        ImageArchiveInvalidException: A file already exists at archive_path, but could not extract an image ID from it.

    :param archive_path: Tar file to read
    :type archive_path: str

    :return: None, if no file at archive_path, or the extracted image ID, which will not have a sha256: prefix.
    :rtype: ImageArchiveManifestSummary
    """
    results = load_archived_image_manifest(archive_path)
    if results is None:
        return None
    if len(results) == 1:
        return results[0]
    raise ImageArchiveInvalidException('Expected to have one entry in manifest.json but found %s' % len(results), None)