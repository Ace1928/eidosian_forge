import functools
import json
import jsonschema
import yaml
from ironicclient import exc
def create_resources(client, filenames):
    """Create resources using their JSON or YAML descriptions.

    :param client: an instance of ironic client;
    :param filenames: a list of filenames containing JSON or YAML resources
        definitions.
    :raises: ClientException if any operation during files processing/resource
        creation fails.
    """
    errors = []
    resources = []
    for resource_file in filenames:
        try:
            resource = load_from_file(resource_file)
            jsonschema.validate(resource, _CREATE_SCHEMA)
            resources.append(resource)
        except (exc.ClientException, jsonschema.ValidationError) as e:
            errors.append(e)
    if errors:
        raise exc.ClientException('While validating the resources file(s), the following error(s) were encountered:\n%s' % '\n'.join((str(e) for e in errors)))
    for r in resources:
        errors.extend(create_chassis(client, r.get('chassis', [])))
        errors.extend(create_nodes(client, r.get('nodes', [])))
    if errors:
        raise exc.ClientException('During resources creation, the following error(s) were encountered:\n%s' % '\n'.join((str(e) for e in errors)))