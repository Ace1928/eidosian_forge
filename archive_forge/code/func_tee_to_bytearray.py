import io
def tee_to_bytearray(response, bytearr, chunksize=_DEFAULT_CHUNKSIZE, decode_content=None):
    """Stream the response both to the generator and a bytearray.

    This will stream the response provided to the function, add them to the
    provided :class:`bytearray` and yield them to the user.

    .. note::

        This uses the :meth:`bytearray.extend` by default instead of passing
        the bytearray into the ``readinto`` method.

    Example usage:

    .. code-block:: python

        b = bytearray()
        resp = requests.get(url, stream=True)
        for chunk in tee_to_bytearray(resp, b):
            # do stuff with chunk

    :param response: Response from requests.
    :type response: requests.Response
    :param bytearray bytearr: Array to add the streamed bytes to.
    :param int chunksize: (optional), Size of chunk to attempt to stream.
    :param bool decode_content: (optional), If True, this will decode the
        compressed content of the response.
    """
    if not isinstance(bytearr, bytearray):
        raise TypeError('tee_to_bytearray() expects bytearr to be a bytearray')
    return _tee(response, bytearr.extend, chunksize, decode_content)