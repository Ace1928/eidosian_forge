from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.facts.facts import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def render_archive(self, root, want):
    archive = want.get('archive')
    archive_node = build_child_xml_node(root, 'archive')
    if 'binary_data' in archive.keys() and archive.get('binary_data'):
        build_child_xml_node(archive_node, 'binary-data')
    if 'files' in archive.keys():
        build_child_xml_node(archive_node, 'files', archive.get('files'))
    if 'no_binary_data' in archive.keys() and archive.get('no_binary_data'):
        build_child_xml_node(archive_node, 'no-binary-data')
    if 'file_size' in archive.keys():
        build_child_xml_node(archive_node, 'size', archive.get('file_size'))
    if 'world_readable' in archive.keys() and archive.get('world_readable'):
        build_child_xml_node(archive_node, 'world-readable')
    if 'no_world_readable' in archive.keys() and archive.get('no_world_readable'):
        build_child_xml_node(archive_node, 'no-world-readable')