def find_plugin_formatters():
    for entrypoint in iter_entry_points(FORMATTER_ENTRY_POINT):
        yield (entrypoint.name, entrypoint.load())