import asyncio
import os
import threading
from threading import Event
from typing import Optional
import discord
import gradio as gr
from discord import Permissions
from discord.ext import commands
from discord.utils import oauth_url
import gradio_client as grc
from gradio_client.utils import QueueError
def run_bot():
    if not DISCORD_TOKEN:
        print('DISCORD_TOKEN NOT SET')
        event.set()
    else:
        bot.run(DISCORD_TOKEN)