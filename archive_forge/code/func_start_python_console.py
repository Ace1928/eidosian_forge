from functools import wraps
def start_python_console(namespace=None, banner='', shells=None):
    """Start Python console bound to the given namespace.
    Readline support and tab completion will be used on Unix, if available.
    """
    if namespace is None:
        namespace = {}
    try:
        shell = get_shell_embed_func(shells)
        if shell is not None:
            shell(namespace=namespace, banner=banner)
    except SystemExit:
        pass