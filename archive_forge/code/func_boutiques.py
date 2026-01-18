import click
from .instance import list_interfaces
from .utils import (
from .. import __version__
@convert.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--interface', type=str, required=True, help='Name of the Nipype interface to export.')
@click.option('-m', '--module', type=PythonModule(), required=True, callback=check_not_none, help='Module where the interface is defined.')
@click.option('-o', '--output', type=UnexistingFilePath, required=True, callback=check_not_none, help='JSON file name where the Boutiques descriptor will be written.')
@click.option('-c', '--container-image', required=True, type=str, help='Name of the container image where the tool is installed.')
@click.option('-p', '--container-type', required=True, type=str, help='Type of container image (Docker or Singularity).')
@click.option('-x', '--container-index', type=str, help='Optional index where the image is available (e.g. http://index.docker.io).')
@click.option('-g', '--ignore-inputs', type=str, multiple=True, help='List of interface inputs to not include in the descriptor.')
@click.option('-v', '--verbose', is_flag=True, flag_value=True, help='Print information messages.')
@click.option('-a', '--author', type=str, help='Author of the tool (required for publishing).')
@click.option('-t', '--tags', type=str, help='JSON string containing tags to include in the descriptor,e.g. "{"key1": "value1"}"')
def boutiques(module, interface, container_image, container_type, output, container_index, verbose, author, ignore_inputs, tags):
    """Nipype to Boutiques exporter.

    See Boutiques specification at https://github.com/boutiques/schema.
    """
    from nipype.utils.nipype2boutiques import generate_boutiques_descriptor
    generate_boutiques_descriptor(module, interface, container_image, container_type, container_index, verbose, True, output, author, ignore_inputs, tags)