from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
from ansible.module_utils.common.text.converters import to_native
def load_archived_image_manifest(archive_path):
    """
    Attempts to get image IDs and image names from metadata stored in the image
    archive tar file.

    The tar should contain a file "manifest.json" with an array with one or more entries,
    and every entry should have a Config field with the image ID in its file name, as
    well as a RepoTags list, which typically has only one entry.

    :raises:
        ImageArchiveInvalidException: A file already exists at archive_path, but could not extract an image ID from it.

    :param archive_path: Tar file to read
    :type archive_path: str

    :return: None, if no file at archive_path, or a list of ImageArchiveManifestSummary objects.
    :rtype: ImageArchiveManifestSummary
    """
    try:
        if not os.path.isfile(archive_path):
            return None
        tf = tarfile.open(archive_path, 'r')
        try:
            try:
                ef = tf.extractfile('manifest.json')
                try:
                    text = ef.read().decode('utf-8')
                    manifest = json.loads(text)
                except Exception as exc:
                    raise ImageArchiveInvalidException('Failed to decode and deserialize manifest.json: %s' % to_native(exc), exc)
                finally:
                    ef.close()
                if len(manifest) == 0:
                    raise ImageArchiveInvalidException('Expected to have at least one entry in manifest.json but found none', None)
                result = []
                for index, meta in enumerate(manifest):
                    try:
                        config_file = meta['Config']
                    except KeyError as exc:
                        raise ImageArchiveInvalidException('Failed to get Config entry from {0}th manifest in manifest.json: {1}'.format(index + 1, to_native(exc)), exc)
                    try:
                        image_id = os.path.splitext(config_file)[0]
                    except Exception as exc:
                        raise ImageArchiveInvalidException('Failed to extract image id from config file name %s: %s' % (config_file, to_native(exc)), exc)
                    for prefix in ('blobs/sha256/',):
                        if image_id.startswith(prefix):
                            image_id = image_id[len(prefix):]
                    try:
                        repo_tags = meta['RepoTags']
                    except KeyError as exc:
                        raise ImageArchiveInvalidException('Failed to get RepoTags entry from {0}th manifest in manifest.json: {1}'.format(index + 1, to_native(exc)), exc)
                    result.append(ImageArchiveManifestSummary(image_id=image_id, repo_tags=repo_tags))
                return result
            except ImageArchiveInvalidException:
                raise
            except Exception as exc:
                raise ImageArchiveInvalidException('Failed to extract manifest.json from tar file %s: %s' % (archive_path, to_native(exc)), exc)
        finally:
            tf.close()
    except ImageArchiveInvalidException:
        raise
    except Exception as exc:
        raise ImageArchiveInvalidException('Failed to open tar file %s: %s' % (archive_path, to_native(exc)), exc)