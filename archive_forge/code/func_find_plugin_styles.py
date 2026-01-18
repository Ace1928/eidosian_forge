def find_plugin_styles():
    for entrypoint in iter_entry_points(STYLE_ENTRY_POINT):
        yield (entrypoint.name, entrypoint.load())