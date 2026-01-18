from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
def dataset_config_operation_started_and_status_log(verb, dataset_config_name, operation_id):
    log.status.Print('{} operation for dataset config {} has been successfully started.\nTo check the status of this operation run: gcloud storage insights operations describe {}'.format(verb, dataset_config_name, operation_id))