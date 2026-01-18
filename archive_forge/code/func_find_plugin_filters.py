def find_plugin_filters():
    for entrypoint in iter_entry_points(FILTER_ENTRY_POINT):
        yield (entrypoint.name, entrypoint.load())