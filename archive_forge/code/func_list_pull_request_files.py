from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def list_pull_request_files(self, pr_number: int) -> List[Dict[str, Any]]:
    """Fetches the full text of all files in a PR. Truncates after first 3k tokens.
        # TODO: Enhancement to summarize files with ctags if they're getting long.

        Args:
            pr_number(int): The number of the pull request on Github

        Returns:
            dict: A dictionary containing the issue's title,
            body, and comments as a string
        """
    tiktoken = _import_tiktoken()
    MAX_TOKENS_FOR_FILES = 3000
    pr_files = []
    pr = self.github_repo_instance.get_pull(number=int(pr_number))
    total_tokens = 0
    page = 0
    while True:
        files_page = pr.get_files().get_page(page)
        if len(files_page) == 0:
            break
        for file in files_page:
            try:
                file_metadata_response = requests.get(file.contents_url)
                if file_metadata_response.status_code == 200:
                    download_url = json.loads(file_metadata_response.text)['download_url']
                else:
                    print(f'Failed to download file: {file.contents_url}, skipping')
                    continue
                file_content_response = requests.get(download_url)
                if file_content_response.status_code == 200:
                    file_content = file_content_response.text
                else:
                    print(f'Failed downloading file content (Error {file_content_response.status_code}). Skipping')
                    continue
                file_tokens = len(tiktoken.get_encoding('cl100k_base').encode(file_content + file.filename + 'file_name file_contents'))
                if total_tokens + file_tokens < MAX_TOKENS_FOR_FILES:
                    pr_files.append({'filename': file.filename, 'contents': file_content, 'additions': file.additions, 'deletions': file.deletions})
                    total_tokens += file_tokens
            except Exception as e:
                print(f'Error when reading files from a PR on github. {e}')
        page += 1
    return pr_files