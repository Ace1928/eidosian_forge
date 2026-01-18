import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def ex_execute_cmd_on_container(self, cont_id, command, **config):
    """
        Description: run a remote command
        Operation: async

        Return: Depends on the  the configuration

        if wait-for-websocket=true and interactive=false
        returns a LXDContainerExecuteResult with:
            uuid=uuid,
            secret_0=fds["0"],
            secret_1=fds["1"],
            secret_2=fds["2"],
            control=fds["control"],
            output={}, result=None

        if wait-for-websocket=true and interactive=true
        returns a LXDContainerExecuteResult with:
            uuid=uuid,
            secret_0=fds["0"],
            secret_1=None,
            secret_2=None,
            control=fds["control"],
            output={}, result=None

        if interactive=false and record-output=true
        returns a LXDContainerExecuteResult with:
            uuid=uuid,
            secret_0=None,
            secret_1=None,
            secret_2=None,
            control=None,
            output=output, result=result

        if none of the above it assumes that the command has
        been executed and returns LXDContainerExecuteResult with:
            uuid=uuid,
            secret_0=None,
            secret_1=None,
            secret_2=None,
            control=None,
            output=None, result=result


        in all the above uuid is the operation id

        :param cont_id: The container name to run the commands
        ":type cont_id: ``str``

        :param command: a list of strings indicating the commands
        and their arguments e.g: ["/bin/bash ls -l"]
        :type  command ``list``

        :param config: Dict with extra arguments.

            For example:

            width:  Initial width of the terminal default 80
            height: Initial height of the terminal default 25
            user:   User to run the command as default 1000
            group: Group to run the  command as default 1000
            cwd: Current working directory default /tmp

            wait-for-websocket: Whether to wait for a connection
            before starting the process. Default False

            record-output: Whether to store stdout and stderr
            (only valid with wait-for-websocket=false)
            (requires API extension container_exec_recording). Default False

            interactive: Whether to allocate a pts device
            instead of PIPEs. Default true

        :type config ``dict``

        :rtype LXDContainerExecuteResult
        """
    input = {'command': command}
    input = LXDContainerDriver._create_exec_configuration(input, **config)
    data = json.dumps(input)
    req = '/{}/containers/{}/exec'.format(self.version, cont_id)
    response = self.connection.request(req, method='POST', data=data)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=100)
    fds = response_dict['metadata']['metadata']['fds']
    uuid = response_dict['metadata']['id']
    if input['wait-for-websocket'] is True and input['interactive'] is False:
        return LXDContainerExecuteResult(uuid=uuid, secret_0=fds['0'], secret_1=fds['1'], secret_2=fds['2'], control=fds['control'], output={}, result=None)
    elif input['wait-for-websocket'] is True and input['interactive'] is True:
        return LXDContainerExecuteResult(uuid=uuid, secret_0=fds['0'], secret_1=None, secret_2=None, control=fds['control'], output={}, result=None)
    elif input['interactive'] is False and input['record-output'] is True:
        output = response_dict['metadata']['metadata']['output']
        result = response_dict['metadata']['metadata']['result']
        return LXDContainerExecuteResult(uuid=uuid, secret_0=None, secret_1=None, secret_2=None, control=None, output=output, result=result)
    else:
        result = response_dict['metadata']['metadata']['result']
        return LXDContainerExecuteResult(uuid=uuid, secret_0=None, secret_1=None, secret_2=None, control=None, output={}, result=result)