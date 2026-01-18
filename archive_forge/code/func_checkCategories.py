from xdg.IniFile import IniFile, is_ascii
import xdg.Locale
from xdg.Exceptions import ParsingError
from xdg.util import which
import os.path
import re
import warnings
def checkCategories(self, value):
    values = self.getList(value)
    main = ['AudioVideo', 'Audio', 'Video', 'Development', 'Education', 'Game', 'Graphics', 'Network', 'Office', 'Science', 'Settings', 'System', 'Utility']
    if not any((item in main for item in values)):
        self.errors.append('Missing main category')
    additional = ['Building', 'Debugger', 'IDE', 'GUIDesigner', 'Profiling', 'RevisionControl', 'Translation', 'Calendar', 'ContactManagement', 'Database', 'Dictionary', 'Chart', 'Email', 'Finance', 'FlowChart', 'PDA', 'ProjectManagement', 'Presentation', 'Spreadsheet', 'WordProcessor', '2DGraphics', 'VectorGraphics', 'RasterGraphics', '3DGraphics', 'Scanning', 'OCR', 'Photography', 'Publishing', 'Viewer', 'TextTools', 'DesktopSettings', 'HardwareSettings', 'Printing', 'PackageManager', 'Dialup', 'InstantMessaging', 'Chat', 'IRCClient', 'Feed', 'FileTransfer', 'HamRadio', 'News', 'P2P', 'RemoteAccess', 'Telephony', 'TelephonyTools', 'VideoConference', 'WebBrowser', 'WebDevelopment', 'Midi', 'Mixer', 'Sequencer', 'Tuner', 'TV', 'AudioVideoEditing', 'Player', 'Recorder', 'DiscBurning', 'ActionGame', 'AdventureGame', 'ArcadeGame', 'BoardGame', 'BlocksGame', 'CardGame', 'KidsGame', 'LogicGame', 'RolePlaying', 'Shooter', 'Simulation', 'SportsGame', 'StrategyGame', 'Art', 'Construction', 'Music', 'Languages', 'ArtificialIntelligence', 'Astronomy', 'Biology', 'Chemistry', 'ComputerScience', 'DataVisualization', 'Economy', 'Electricity', 'Geography', 'Geology', 'Geoscience', 'History', 'Humanities', 'ImageProcessing', 'Literature', 'Maps', 'Math', 'NumericalAnalysis', 'MedicalSoftware', 'Physics', 'Robotics', 'Spirituality', 'Sports', 'ParallelComputing', 'Amusement', 'Archiving', 'Compression', 'Electronics', 'Emulator', 'Engineering', 'FileTools', 'FileManager', 'TerminalEmulator', 'Filesystem', 'Monitor', 'Security', 'Accessibility', 'Calculator', 'Clock', 'TextEditor', 'Documentation', 'Adult', 'Core', 'KDE', 'GNOME', 'XFCE', 'GTK', 'Qt', 'Motif', 'Java', 'ConsoleOnly']
    allcategories = additional + main
    for item in values:
        if item not in allcategories and (not item.startswith('X-')):
            self.errors.append("'%s' is not a registered Category" % item)