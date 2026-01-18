from __future__ import annotations
import collections.abc as c
import dataclasses
import os
import typing as t
from .util import (
from .provider import (
from .provider.source import (
from .provider.source.unversioned import (
from .provider.source.installed import (
from .provider.source.unsupported import (
from .provider.layout import (
from .provider.layout.unsupported import (
def create_collection_layouts(self) -> list[ContentLayout]:
    """
        Return a list of collection layouts, one for each collection in the same collection root as the current collection layout.
        An empty list is returned if the current content layout is not a collection layout.
        """
    layout = self.content
    collection = layout.collection
    if not collection:
        return []
    root_path = os.path.join(collection.root, 'ansible_collections')
    display.info('Scanning collection root: %s' % root_path, verbosity=1)
    namespace_names = sorted((name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))))
    collections = []
    for namespace_name in namespace_names:
        namespace_path = os.path.join(root_path, namespace_name)
        collection_names = sorted((name for name in os.listdir(namespace_path) if os.path.isdir(os.path.join(namespace_path, name))))
        for collection_name in collection_names:
            collection_path = os.path.join(namespace_path, collection_name)
            if collection_path == os.path.join(collection.root, collection.directory):
                collection_layout = layout
            else:
                collection_layout = self.__create_content_layout(self.__layout_providers, self.__source_providers, collection_path, False)[0]
            file_count = len(collection_layout.all_files())
            if not file_count:
                continue
            display.info('Including collection: %s (%d files)' % (collection_layout.collection.full_name, file_count), verbosity=1)
            collections.append(collection_layout)
    return collections