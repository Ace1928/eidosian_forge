import stat
from ... import controldir
def open_destination_directory(location, format=None, verbose=True):
    """Open a destination directory and return the BzrDir.

    If destination has a control directory, it will be returned.
    Otherwise, the destination should be empty or non-existent and
    a shared repository will be created there.

    :param location: the destination directory
    :param format: the format to use or None for the default
    :param verbose: display the format used if a repository is created.
    :return: BzrDir for the destination
    """
    import os
    from ... import controldir, errors, trace, transport
    try:
        control, relpath = controldir.ControlDir.open_containing(location)
        return control
    except errors.NotBranchError:
        pass
    if os.path.exists(location):
        contents = os.listdir(location)
        if contents:
            errors.CommandError('Destination must have a .bzr directory,  not yet exist or be empty - files found in %s' % (location,))
    else:
        try:
            os.mkdir(location)
        except OSError as ex:
            raise errors.CommandError('Unable to create {}: {}'.format(location, ex))
    trace.note('Creating destination repository ...')
    if format is None:
        format = controldir.format_registry.make_controldir('default')
    to_transport = transport.get_transport(location)
    to_transport.ensure_base()
    control = format.initialize_on_transport(to_transport)
    repo = control.create_repository(shared=True)
    if verbose:
        from ...info import show_bzrdir_info
        show_bzrdir_info(repo.controldir, verbose=0)
    return control