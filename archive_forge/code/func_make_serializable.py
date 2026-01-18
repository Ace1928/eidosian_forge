from googlecloudsdk.core import resource
def make_serializable(api_response, _):
    """Serializes the given API response.

  Args:
    api_response: the api response.
  Returns:
    the serialized api response.
  """
    return resource.resource_projector.MakeSerializable(api_response)