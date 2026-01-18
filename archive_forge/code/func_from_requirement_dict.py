from __future__ import (absolute_import, division, print_function)
import os
import typing as t
from collections import namedtuple
from collections.abc import MutableSequence, MutableMapping
from glob import iglob
from urllib.parse import urlparse
from yaml import safe_load
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection import HAS_PACKAGING, PkgReq
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
@classmethod
def from_requirement_dict(cls, collection_req, art_mgr, validate_signature_options=True):
    req_name = collection_req.get('name', None)
    req_version = collection_req.get('version', '*')
    req_type = collection_req.get('type')
    req_source = collection_req.get('source', None)
    req_signature_sources = collection_req.get('signatures', None)
    if req_signature_sources is not None:
        if validate_signature_options and art_mgr.keyring is None:
            raise AnsibleError(f'Signatures were provided to verify {req_name} but no keyring was configured.')
        if not isinstance(req_signature_sources, MutableSequence):
            req_signature_sources = [req_signature_sources]
        req_signature_sources = frozenset(req_signature_sources)
    if req_type is None:
        if _ALLOW_CONCRETE_POINTER_IN_SOURCE and req_source is not None and _is_concrete_artifact_pointer(req_source):
            src_path = req_source
        elif req_name is not None and AnsibleCollectionRef.is_valid_collection_name(req_name):
            req_type = 'galaxy'
        elif req_name is not None and _is_concrete_artifact_pointer(req_name):
            src_path, req_name = (req_name, None)
        else:
            dir_tip_tmpl = '\n\nTip: Make sure you are pointing to the right subdirectory — `{src!s}` looks like a directory but it is neither a collection, nor a namespace dir.'
            if req_source is not None and os.path.isdir(req_source):
                tip = dir_tip_tmpl.format(src=req_source)
            elif req_name is not None and os.path.isdir(req_name):
                tip = dir_tip_tmpl.format(src=req_name)
            elif req_name:
                tip = '\n\nCould not find {0}.'.format(req_name)
            else:
                tip = ''
            raise AnsibleError("Neither the collection requirement entry key 'name', nor 'source' point to a concrete resolvable collection artifact. Also 'name' is not an FQCN. A valid collection name must be in the format <namespace>.<collection>. Please make sure that the namespace and the collection name contain characters from [a-zA-Z0-9_] only.{extra_tip!s}".format(extra_tip=tip))
    if req_type is None:
        if _is_git_url(src_path):
            req_type = 'git'
            req_source = src_path
        elif _is_http_url(src_path):
            req_type = 'url'
            req_source = src_path
        elif _is_file_path(src_path):
            req_type = 'file'
            req_source = src_path
        elif _is_collection_dir(src_path):
            if _is_installed_collection_dir(src_path) and _is_collection_src_dir(src_path):
                raise AnsibleError(u"Collection requirement at '{path!s}' has both a {manifest_json!s} file and a {galaxy_yml!s}.\nThe requirement must either be an installed collection directory or a source collection directory, not both.".format(path=to_text(src_path, errors='surrogate_or_strict'), manifest_json=to_text(_MANIFEST_JSON), galaxy_yml=to_text(_GALAXY_YAML)))
            req_type = 'dir'
            req_source = src_path
        elif _is_collection_namespace_dir(src_path):
            req_name = None
            req_type = 'subdirs'
            req_source = src_path
        else:
            raise AnsibleError('Failed to automatically detect the collection requirement type.')
    if req_type not in {'file', 'galaxy', 'git', 'url', 'dir', 'subdirs'}:
        raise AnsibleError("The collection requirement entry key 'type' must be one of file, galaxy, git, dir, subdirs, or url.")
    if req_name is None and req_type == 'galaxy':
        raise AnsibleError("Collections requirement entry should contain the key 'name' if it's requested from a Galaxy-like index server.")
    if req_type != 'galaxy' and req_source is None:
        req_source, req_name = (req_name, None)
    if req_type == 'galaxy' and isinstance(req_source, GalaxyAPI) and (not _is_http_url(req_source.api_server)):
        raise AnsibleError("Collections requirement 'source' entry should contain a valid Galaxy API URL but it does not: {not_url!s} is not an HTTP URL.".format(not_url=req_source.api_server))
    if req_type == 'dir' and req_source.endswith(os.path.sep):
        req_source = req_source.rstrip(os.path.sep)
    tmp_inst_req = cls(req_name, req_version, req_source, req_type, req_signature_sources)
    if req_type not in {'galaxy', 'subdirs'} and req_name is None:
        req_name = art_mgr.get_direct_collection_fqcn(tmp_inst_req)
    if req_type not in {'galaxy', 'subdirs'} and req_version == '*':
        req_version = art_mgr.get_direct_collection_version(tmp_inst_req)
    return cls(req_name, req_version, req_source, req_type, req_signature_sources)