from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@classmethod
def from_secrets(cls, access_token: str, access_token_secret: str, consumer_key: str, consumer_secret: str, twitter_users: Sequence[str], number_tweets: Optional[int]=100) -> TwitterTweetLoader:
    """Create a TwitterTweetLoader from access tokens and secrets."""
    tweepy = _dependable_tweepy_import()
    auth = tweepy.OAuthHandler(access_token=access_token, access_token_secret=access_token_secret, consumer_key=consumer_key, consumer_secret=consumer_secret)
    return cls(auth_handler=auth, twitter_users=twitter_users, number_tweets=number_tweets)