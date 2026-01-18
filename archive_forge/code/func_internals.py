from docutils import core, io
def internals(input_string, source_path=None, destination_path=None, input_encoding='unicode', settings_overrides=None):
    """
    Return the document tree and publisher, for exploring Docutils internals.

    Parameters: see `html_parts()`.
    """
    if settings_overrides:
        overrides = settings_overrides.copy()
    else:
        overrides = {}
    overrides['input_encoding'] = input_encoding
    output, pub = core.publish_programmatically(source_class=io.StringInput, source=input_string, source_path=source_path, destination_class=io.NullOutput, destination=None, destination_path=destination_path, reader=None, reader_name='standalone', parser=None, parser_name='restructuredtext', writer=None, writer_name='null', settings=None, settings_spec=None, settings_overrides=overrides, config_section=None, enable_exit_status=None)
    return (pub.writer.document, pub)