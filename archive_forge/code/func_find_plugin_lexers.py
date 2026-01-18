def find_plugin_lexers():
    for entrypoint in iter_entry_points(LEXER_ENTRY_POINT):
        yield entrypoint.load()