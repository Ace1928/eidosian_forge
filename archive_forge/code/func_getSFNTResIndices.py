from io import BytesIO
from fontTools.misc.macRes import ResourceReader, ResourceError
def getSFNTResIndices(path):
    """Determine whether a file has a 'sfnt' resource fork or not."""
    try:
        reader = ResourceReader(path)
        indices = reader.getIndices('sfnt')
        reader.close()
        return indices
    except ResourceError:
        return []