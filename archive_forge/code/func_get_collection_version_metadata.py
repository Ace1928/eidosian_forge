from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
def get_collection_version_metadata(self, collection_candidate):
    """Retrieve collection metadata of a given candidate."""
    self._assert_that_offline_mode_is_not_requested()
    api_lookup_order = (collection_candidate.src,) if isinstance(collection_candidate.src, GalaxyAPI) else self._apis
    last_err: t.Optional[Exception]
    for api in api_lookup_order:
        try:
            version_metadata = api.get_collection_version_metadata(collection_candidate.namespace, collection_candidate.name, collection_candidate.ver)
        except GalaxyError as api_err:
            last_err = api_err
        except Exception as unknown_err:
            last_err = unknown_err
            display.warning('Skipping Galaxy server {server!s}. Got an unexpected error when getting available versions of collection {fqcn!s}: {err!s}'.format(server=api.api_server, fqcn=collection_candidate.fqcn, err=to_text(unknown_err)))
        else:
            self._concrete_art_mgr.save_collection_source(collection_candidate, version_metadata.download_url, version_metadata.artifact_sha256, api.token, version_metadata.signatures_url, version_metadata.signatures)
            return version_metadata
    raise last_err