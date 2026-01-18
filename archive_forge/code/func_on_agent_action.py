from typing import Any, Dict, Optional, TextIO, cast
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.input import print_text
def on_agent_action(self, action: AgentAction, color: Optional[str]=None, **kwargs: Any) -> Any:
    """Run on agent action."""
    print_text(action.log, color=color or self.color, file=self.file)