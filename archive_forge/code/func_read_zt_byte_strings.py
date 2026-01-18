def read_zt_byte_strings(fobj, n_strings=1, bufsize=1024):
    """Read zero-terminated byte strings from a file object `fobj`

    Returns byte strings with terminal zero stripped.

    Found strings can be of any length.

    The file position of `fobj` on exit will be at the byte after the terminal
    0 of the final read byte string.

    Parameters
    ----------
    f : fileobj
        File object to use.  Should implement ``read``, returning byte objects,
        and ``seek(n, 1)`` to seek from current file position.
    n_strings : int, optional
        Number of byte strings to return
    bufsize: int, optional
       Define chunk size to load from file while searching for zero terminals.
       We load this many bytes at a time from the file, but the returned
       strings can be longer than `bufsize`.

    Returns
    -------
    byte_strings : list
        List of byte strings, where strings do not include the terminal 0
    """
    byte_strings = []
    trailing = b''
    while True:
        buf = fobj.read(bufsize)
        eof = len(buf) < bufsize
        zt_strings = buf.split(b'\x00')
        if len(zt_strings) > 1:
            byte_strings += [trailing + zt_strings[0]] + zt_strings[1:-1]
            trailing = zt_strings[-1]
        else:
            trailing += zt_strings[0]
        n_found = len(byte_strings)
        if eof or n_found >= n_strings:
            break
    if n_found < n_strings:
        raise ValueError(f'Expected {n_strings} strings, found {n_found}')
    n_extra = n_found - n_strings
    leftover_strings = byte_strings[n_strings:] + [trailing]
    extra_bytes = sum((len(bs) for bs in leftover_strings)) + n_extra
    fobj.seek(-extra_bytes, 1)
    return byte_strings[:n_strings]