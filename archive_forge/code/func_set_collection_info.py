def set_collection_info(collection_name=None, collection_version=None):
    if collection_name:
        _collection_info_context['name'] = collection_name
    if collection_version:
        _collection_info_context['version'] = collection_version