import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def layerContentsValidator(value, ufoPathOrFileSystem):
    """
    Check the validity of layercontents.plist.
    Version 3+.
    """
    if isinstance(ufoPathOrFileSystem, fs.base.FS):
        fileSystem = ufoPathOrFileSystem
    else:
        fileSystem = fs.osfs.OSFS(ufoPathOrFileSystem)
    bogusFileMessage = 'layercontents.plist in not in the correct format.'
    if not isinstance(value, list):
        return (False, bogusFileMessage)
    usedLayerNames = set()
    usedDirectories = set()
    contents = {}
    for entry in value:
        if not isinstance(entry, list):
            return (False, bogusFileMessage)
        if not len(entry) == 2:
            return (False, bogusFileMessage)
        for i in entry:
            if not isinstance(i, str):
                return (False, bogusFileMessage)
        layerName, directoryName = entry
        if directoryName != 'glyphs':
            if not directoryName.startswith('glyphs.'):
                return (False, 'Invalid directory name (%s) in layercontents.plist.' % directoryName)
        if len(layerName) == 0:
            return (False, 'Empty layer name in layercontents.plist.')
        if not fileSystem.exists(directoryName):
            return (False, 'A glyphset does not exist at %s.' % directoryName)
        if layerName == 'public.default' and directoryName != 'glyphs':
            return (False, 'The name public.default is being used by a layer that is not the default.')
        if layerName in usedLayerNames:
            return (False, 'The layer name %s is used by more than one layer.' % layerName)
        usedLayerNames.add(layerName)
        if directoryName in usedDirectories:
            return (False, 'The directory %s is used by more than one layer.' % directoryName)
        usedDirectories.add(directoryName)
        contents[layerName] = directoryName
    foundDefault = 'glyphs' in contents.values()
    if not foundDefault:
        return (False, 'The required default glyph set is not in the UFO.')
    return (True, None)