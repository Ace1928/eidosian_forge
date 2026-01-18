from __future__ import annotations
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def flush_tracker(self, name: Optional[str]=None, langchain_asset: Any=None, finish: bool=False) -> None:
    """Flush the tracker and setup the session.

        Everything after this will be a new table.

        Args:
            name: Name of the performed session so far so it is identifiable
            langchain_asset: The langchain asset to save.
            finish: Whether to finish the run.

            Returns:
                None
        """
    pd = import_pandas()
    clearml = import_clearml()
    self.logger.report_table('Action Records', name, table_plot=pd.DataFrame(self.action_records))
    session_analysis_df = self._create_session_analysis_df()
    self.logger.report_table('Session Analysis', name, table_plot=session_analysis_df)
    if self.stream_logs:
        self.logger.report_text({'action_records': pd.DataFrame(self.action_records), 'session_analysis': session_analysis_df})
    if langchain_asset:
        langchain_asset_path = Path(self.temp_dir.name, 'model.json')
        try:
            langchain_asset.save(langchain_asset_path)
            output_model = clearml.OutputModel(task=self.task, config_text=load_json(langchain_asset_path))
            output_model.update_weights(weights_filename=str(langchain_asset_path), auto_delete_file=False, target_filename=name)
        except ValueError:
            langchain_asset.save_agent(langchain_asset_path)
            output_model = clearml.OutputModel(task=self.task, config_text=load_json(langchain_asset_path))
            output_model.update_weights(weights_filename=str(langchain_asset_path), auto_delete_file=False, target_filename=name)
        except NotImplementedError as e:
            print('Could not save model.')
            print(repr(e))
            pass
    self.task.flush(wait_for_uploads=True)
    self.temp_dir.cleanup()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.reset_callback_meta()
    if finish:
        self.task.close()