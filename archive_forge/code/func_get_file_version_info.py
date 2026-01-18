from __future__ import with_statement
from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window
import sys
import os
import ctypes
import warnings
from os import path, getenv
@classmethod
def get_file_version_info(cls, filename):
    """
        Get the program version from an executable file, if available.

        @type  filename: str
        @param filename: Pathname to the executable file to query.

        @rtype: tuple(str, str, bool, bool, str, str)
        @return: Tuple with version information extracted from the executable
            file metadata, containing the following:
             - File version number (C{"major.minor"}).
             - Product version number (C{"major.minor"}).
             - C{True} for debug builds, C{False} for production builds.
             - C{True} for legacy OS builds (DOS, OS/2, Win16),
               C{False} for modern OS builds.
             - Binary file type.
               May be one of the following values:
                - "application"
                - "dynamic link library"
                - "static link library"
                - "font"
                - "raster font"
                - "TrueType font"
                - "vector font"
                - "driver"
                - "communications driver"
                - "display driver"
                - "installable driver"
                - "keyboard driver"
                - "language driver"
                - "legacy driver"
                - "mouse driver"
                - "network driver"
                - "printer driver"
                - "sound driver"
                - "system driver"
                - "versioned printer driver"
             - Binary creation timestamp.
            Any of the fields may be C{None} if not available.

        @raise WindowsError: Raises an exception on error.
        """
    pBlock = win32.GetFileVersionInfo(filename)
    pBuffer, dwLen = win32.VerQueryValue(pBlock, '\\')
    if dwLen != ctypes.sizeof(win32.VS_FIXEDFILEINFO):
        raise ctypes.WinError(win32.ERROR_BAD_LENGTH)
    pVersionInfo = ctypes.cast(pBuffer, ctypes.POINTER(win32.VS_FIXEDFILEINFO))
    VersionInfo = pVersionInfo.contents
    if VersionInfo.dwSignature != 4277077181:
        raise ctypes.WinError(win32.ERROR_BAD_ARGUMENTS)
    FileVersion = '%d.%d' % (VersionInfo.dwFileVersionMS, VersionInfo.dwFileVersionLS)
    ProductVersion = '%d.%d' % (VersionInfo.dwProductVersionMS, VersionInfo.dwProductVersionLS)
    if VersionInfo.dwFileFlagsMask & win32.VS_FF_DEBUG:
        DebugBuild = VersionInfo.dwFileFlags & win32.VS_FF_DEBUG != 0
    else:
        DebugBuild = None
    LegacyBuild = VersionInfo.dwFileOS != win32.VOS_NT_WINDOWS32
    FileType = cls.__binary_types.get(VersionInfo.dwFileType)
    if VersionInfo.dwFileType == win32.VFT_DRV:
        FileType = cls.__driver_types.get(VersionInfo.dwFileSubtype)
    elif VersionInfo.dwFileType == win32.VFT_FONT:
        FileType = cls.__font_types.get(VersionInfo.dwFileSubtype)
    FileDate = (VersionInfo.dwFileDateMS << 32) + VersionInfo.dwFileDateLS
    if FileDate:
        CreationTime = win32.FileTimeToSystemTime(FileDate)
        CreationTimestamp = '%s, %s %d, %d (%d:%d:%d.%d)' % (cls.__days_of_the_week[CreationTime.wDayOfWeek], cls.__months[CreationTime.wMonth], CreationTime.wDay, CreationTime.wYear, CreationTime.wHour, CreationTime.wMinute, CreationTime.wSecond, CreationTime.wMilliseconds)
    else:
        CreationTimestamp = None
    return (FileVersion, ProductVersion, DebugBuild, LegacyBuild, FileType, CreationTimestamp)