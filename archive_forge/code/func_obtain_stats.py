from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.dict import DictionaryAgent
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
def obtain_stats(opt, parser):
    report_text, report_log = verify(opt)
    print(report_text.replace('\\n', '\n'))