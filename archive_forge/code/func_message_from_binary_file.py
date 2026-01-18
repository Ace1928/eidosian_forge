def message_from_binary_file(fp, *args, **kws):
    """Read a binary file and parse its contents into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from email.parser import BytesParser
    return BytesParser(*args, **kws).parse(fp)