from pathlib import Path
def path_type_label(p: Path) -> str:
    """
    Find out what sort of thing a path is.
    """
    assert p.exists(), 'path does not exist'
    for method, name in path_type_labels.items():
        if getattr(p, method)():
            return name
    return 'unknown'