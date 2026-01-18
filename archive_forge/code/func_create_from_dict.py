import re
from os import path
from ruamel import yaml
from kubernetes import client
def create_from_dict(k8s_client, data, verbose=False, namespace='default', **kwargs):
    """
    Perform an action from a dictionary containing valid kubernetes
    API object (i.e. List, Service, etc).

    Input:
    k8s_client: an ApiClient object, initialized with the client args.
    data: a dictionary holding valid kubernetes objects
    verbose: If True, print confirmation from the create action.
        Default is False.
    namespace: string. Contains the namespace to create all
        resources inside. The namespace must preexist otherwise
        the resource creation will fail. If the API object in
        the yaml file already contains a namespace definition
        this parameter has no effect.

    Raises:
        FailToCreateError which holds list of `client.rest.ApiException`
        instances for each object that failed to create.
    """
    api_exceptions = []
    if 'List' in data['kind']:
        kind = data['kind'].replace('List', '')
        for yml_object in data['items']:
            if kind != '':
                yml_object['apiVersion'] = data['apiVersion']
                yml_object['kind'] = kind
            try:
                create_from_yaml_single_item(k8s_client, yml_object, verbose, namespace=namespace, **kwargs)
            except client.rest.ApiException as api_exception:
                api_exceptions.append(api_exception)
    else:
        try:
            create_from_yaml_single_item(k8s_client, data, verbose, namespace=namespace, **kwargs)
        except client.rest.ApiException as api_exception:
            api_exceptions.append(api_exception)
    if api_exceptions:
        raise FailToCreateError(api_exceptions)