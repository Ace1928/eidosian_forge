from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
def getModifiers(nsevent):
    modifiers = 0
    modifierFlags = nsevent.modifierFlags()
    if modifierFlags & cocoapy.NSAlphaShiftKeyMask:
        modifiers |= key.MOD_CAPSLOCK
    if modifierFlags & cocoapy.NSShiftKeyMask:
        modifiers |= key.MOD_SHIFT
    if modifierFlags & cocoapy.NSControlKeyMask:
        modifiers |= key.MOD_CTRL
    if modifierFlags & cocoapy.NSAlternateKeyMask:
        modifiers |= key.MOD_ALT
        modifiers |= key.MOD_OPTION
    if modifierFlags & cocoapy.NSCommandKeyMask:
        modifiers |= key.MOD_COMMAND
    if modifierFlags & cocoapy.NSFunctionKeyMask:
        modifiers |= key.MOD_FUNCTION
    return modifiers