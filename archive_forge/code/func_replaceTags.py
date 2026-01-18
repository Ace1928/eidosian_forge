from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.artifacts.vulnerabilities import GetLatestScan
from googlecloudsdk.api_lib.artifacts.vulnerabilities import GetVulnerabilities
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import format_util
def replaceTags(self, original_uri):
    updated_uri = original_uri
    if not updated_uri.startswith('https://'):
        updated_uri = 'https://{}'.format(updated_uri)
    found = re.findall(docker_util.DOCKER_URI_REGEX, updated_uri)
    if found:
        resource_uri_str = found[0][2]
        image, version = docker_util.DockerUrlToVersion(resource_uri_str)
        project = image.project
        docker_html_str_digest = 'https://{}'.format(version.GetDockerString())
        updated_uri = re.sub(docker_util.DOCKER_URI_REGEX, docker_html_str_digest, updated_uri, 1)
        return (updated_uri, project)
    raise ar_exceptions.InvalidInputValueError('Received invalid URI {}'.format(original_uri))