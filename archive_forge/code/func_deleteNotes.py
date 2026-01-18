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
def deleteNotes(self, file_notes, project, uri):
    list_request = self.ca_messages.ContaineranalysisProjectsNotesListRequest(filter='vulnerability_assessment.product.generic_uri="{}"'.format(uri), parent='projects/{}'.format(project))
    db_notes = list_pager.YieldFromList(service=self.ca_client.projects_notes, request=list_request, field='notes', batch_size_attribute='pageSize')
    cves_in_file = set()
    for file_note in file_notes:
        file_uri = file_note.value.vulnerabilityAssessment.product.genericUri
        file_vulnerability = file_note.value.vulnerabilityAssessment.assessment.vulnerabilityId
        if file_uri == uri:
            cves_in_file.add(file_vulnerability)
    for db_note in db_notes:
        db_vulnerability = db_note.vulnerabilityAssessment.assessment.vulnerabilityId
        if db_vulnerability not in cves_in_file:
            delete_request = self.ca_messages.ContaineranalysisProjectsNotesDeleteRequest(name=db_note.name)
            self.ca_client.projects_notes.Delete(delete_request)