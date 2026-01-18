from .. import errors
from .. import utils
from ..types import CancellableStream
@utils.check_resource('exec_id')
def exec_start(self, exec_id, detach=False, tty=False, stream=False, socket=False, demux=False):
    """
        Start a previously set up exec instance.

        Args:
            exec_id (str): ID of the exec instance
            detach (bool): If true, detach from the exec command.
                Default: False
            tty (bool): Allocate a pseudo-TTY. Default: False
            stream (bool): Return response data progressively as an iterator
                of strings, rather than a single string.
            socket (bool): Return the connection socket to allow custom
                read/write operations. Must be closed by the caller when done.
            demux (bool): Return stdout and stderr separately

        Returns:

            (generator or str or tuple): If ``stream=True``, a generator
            yielding response chunks. If ``socket=True``, a socket object for
            the connection. A string containing response data otherwise. If
            ``demux=True``, a tuple with two elements of type byte: stdout and
            stderr.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    data = {'Tty': tty, 'Detach': detach}
    headers = {} if detach else {'Connection': 'Upgrade', 'Upgrade': 'tcp'}
    res = self._post_json(self._url('/exec/{0}/start', exec_id), headers=headers, data=data, stream=True)
    if detach:
        try:
            return self._result(res)
        finally:
            res.close()
    if socket:
        return self._get_raw_response_socket(res)
    output = self._read_from_socket(res, stream, tty=tty, demux=demux)
    if stream:
        return CancellableStream(output, res)
    else:
        return output