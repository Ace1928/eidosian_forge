import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
@classmethod
def required_fields(cls):
    return {'secret': 'The name of the secret in Secret Manager. Must be a secret in the same project being deployed or be an alias mapped in the `run.googleapis.com/secrets` annotation.', 'version': 'The version of the secret to make available in the volume.', 'path': 'The relative path within the volume to mount that version.'}