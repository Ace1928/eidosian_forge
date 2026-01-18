from __future__ import annotations
import urllib.parse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NamedTuple
import github.Authorization
import github.Event
import github.Gist
import github.GithubObject
import github.Invitation
import github.Issue
import github.Membership
import github.Migration
import github.NamedUser
import github.Notification
import github.Organization
import github.Plan
import github.Repository
import github.UserKey
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def create_repo(self, name: str, description: Opt[str]=NotSet, homepage: Opt[str]=NotSet, private: Opt[bool]=NotSet, has_issues: Opt[bool]=NotSet, has_wiki: Opt[bool]=NotSet, has_downloads: Opt[bool]=NotSet, has_projects: Opt[bool]=NotSet, auto_init: Opt[bool]=NotSet, license_template: Opt[str]=NotSet, gitignore_template: Opt[str]=NotSet, allow_squash_merge: Opt[bool]=NotSet, allow_merge_commit: Opt[bool]=NotSet, allow_rebase_merge: Opt[bool]=NotSet, delete_branch_on_merge: Opt[bool]=NotSet) -> Repository:
    """
        :calls: `POST /user/repos <http://docs.github.com/en/rest/reference/repos>`_
        """
    assert isinstance(name, str), name
    assert is_optional(description, str), description
    assert is_optional(homepage, str), homepage
    assert is_optional(private, bool), private
    assert is_optional(has_issues, bool), has_issues
    assert is_optional(has_wiki, bool), has_wiki
    assert is_optional(has_downloads, bool), has_downloads
    assert is_optional(has_projects, bool), has_projects
    assert is_optional(auto_init, bool), auto_init
    assert is_optional(license_template, str), license_template
    assert is_optional(gitignore_template, str), gitignore_template
    assert is_optional(allow_squash_merge, bool), allow_squash_merge
    assert is_optional(allow_merge_commit, bool), allow_merge_commit
    assert is_optional(allow_rebase_merge, bool), allow_rebase_merge
    assert is_optional(delete_branch_on_merge, bool), delete_branch_on_merge
    post_parameters: dict[str, Any] = NotSet.remove_unset_items({'name': name, 'description': description, 'homepage': homepage, 'private': private, 'has_issues': has_issues, 'has_wiki': has_wiki, 'has_downloads': has_downloads, 'has_projects': has_projects, 'auto_init': auto_init, 'license_template': license_template, 'gitignore_template': gitignore_template, 'allow_squash_merge': allow_squash_merge, 'allow_merge_commit': allow_merge_commit, 'allow_rebase_merge': allow_rebase_merge, 'delete_branch_on_merge': delete_branch_on_merge})
    headers, data = self._requester.requestJsonAndCheck('POST', '/user/repos', input=post_parameters)
    return github.Repository.Repository(self._requester, headers, data, completed=True)