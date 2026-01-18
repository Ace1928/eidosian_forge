import re
from os import path
from ruamel import yaml
from kubernetes import client
def create_from_yaml(k8s_client, yaml_file, verbose=False, namespace='default', **kwargs):
    """
    Perform an action from a yaml file. Pass True for verbose to
    print confirmation information.
    Input:
    yaml_file: string. Contains the path to yaml file.
    k8s_client: an ApiClient object, initialized with the client args.
    verbose: If True, print confirmation from the create action.
        Default is False.
    namespace: string. Contains the namespace to create all
        resources inside. The namespace must preexist otherwise
        the resource creation will fail. If the API object in
        the yaml file already contains a namespace definition
        this parameter has no effect.

    Available parameters for creating <kind>:
    :param async_req bool
    :param bool include_uninitialized: If true, partially initialized
        resources are included in the response.
    :param str pretty: If 'true', then the output is pretty printed.
    :param str dry_run: When present, indicates that modifications
        should not be persisted. An invalid or unrecognized dryRun
        directive will result in an error response and no further
        processing of the request.
        Valid values are: - All: all dry run stages will be processed

    Raises:
        FailToCreateError which holds list of `client.rest.ApiException`
        instances for each object that failed to create.
    """
    with open(path.abspath(yaml_file)) as f:
        yml_document_all = yaml.safe_load_all(f)
        failures = []
        for yml_document in yml_document_all:
            try:
                create_from_dict(k8s_client, yml_document, verbose, namespace=namespace, **kwargs)
            except FailToCreateError as failure:
                failures.extend(failure.api_exceptions)
        if failures:
            raise FailToCreateError(failures)