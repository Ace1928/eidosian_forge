from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import vex_util
def batchWriteNotes(self, notes, project):
    if not notes:
        return
    note_value = self.ca_messages.BatchCreateNotesRequest.NotesValue()
    note_value.additionalProperties = notes
    batch_request = self.ca_messages.BatchCreateNotesRequest(notes=note_value)
    request = self.ca_messages.ContaineranalysisProjectsNotesBatchCreateRequest(parent='projects/{}'.format(project), batchCreateNotesRequest=batch_request)
    self.ca_client.projects_notes.BatchCreate(request)