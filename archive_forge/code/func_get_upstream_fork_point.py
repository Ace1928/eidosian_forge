import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
def get_upstream_fork_point(self) -> Any:
    """Get the most recent ancestor of HEAD that occurs on an upstream branch.

        First looks at the current branch's tracking branch, if applicable. If
        that doesn't work, looks at every other branch to find the most recent
        ancestor of HEAD that occurs on a tracking branch.

        Returns:
            git.Commit object or None
        """
    possible_relatives = []
    try:
        if not self.repo:
            return None
        try:
            active_branch = self.repo.active_branch
        except (TypeError, ValueError):
            logger.debug('git is in a detached head state')
            return None
        else:
            tracking_branch = active_branch.tracking_branch()
            if tracking_branch:
                possible_relatives.append(tracking_branch.commit)
        if not possible_relatives:
            for branch in self.repo.branches:
                tracking_branch = branch.tracking_branch()
                if tracking_branch is not None:
                    possible_relatives.append(tracking_branch.commit)
        head = self.repo.head
        most_recent_ancestor = None
        for possible_relative in possible_relatives:
            for ancestor in self.repo.merge_base(head, possible_relative):
                if most_recent_ancestor is None:
                    most_recent_ancestor = ancestor
                elif self.repo.is_ancestor(most_recent_ancestor, ancestor):
                    most_recent_ancestor = ancestor
        return most_recent_ancestor
    except GitCommandError as e:
        logger.debug('git remote upstream fork point could not be found')
        logger.debug(str(e))
        return None