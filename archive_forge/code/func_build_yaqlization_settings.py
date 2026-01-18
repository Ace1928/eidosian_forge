def build_yaqlization_settings(yaqlize_attributes=True, yaqlize_methods=True, yaqlize_indexer=True, auto_yaqlize_result=False, whitelist=None, blacklist=None, attribute_remapping=None, blacklist_remapped_attributes=True):
    whitelist = set(whitelist or [])
    blacklist = set(blacklist or [])
    attribute_remapping = attribute_remapping or {}
    if blacklist_remapped_attributes:
        for value in attribute_remapping.values():
            if not isinstance(value, str):
                name = value[0]
            else:
                name = value
            blacklist.add(name)
    return {'yaqlizeAttributes': yaqlize_attributes, 'yaqlizeMethods': yaqlize_methods, 'yaqlizeIndexer': yaqlize_indexer, 'autoYaqlizeResult': auto_yaqlize_result, 'whitelist': whitelist, 'blacklist': blacklist, 'attributeRemapping': attribute_remapping}