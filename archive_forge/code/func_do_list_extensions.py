from cinderclient import base
from cinderclient import shell_utils
def do_list_extensions(client, _args):
    """Lists all available os-api extensions."""
    extensions = client.list_extensions.show_all()
    fields = ['Name', 'Summary', 'Alias', 'Updated']
    shell_utils.print_list(extensions, fields)